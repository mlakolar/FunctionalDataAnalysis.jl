import KernelSmoothing
import StatsAPI

@testset "Covariance pooling and exact sufficient moments" begin
    FDA = FunctionalDataAnalysis
    @test FDA._pooled_subject_offsets([1,1,2,2,2,3],3) == [1,3,6,7]
    grid = [0.,.2,.5,.8,1.]
    values = [sin(i+t)+i*t+t^2 for i in 1:6, t in grid]
    complete = GridFunctionalData(values,grid)
    masked_values = Matrix{Union{Missing,Float64}}(values)
    masked_values[2,2] = masked_values[4,4] = missing
    masked = GridFunctionalData(masked_values,grid)
    times = [grid[isodd(i) ? [1,2,3,5] : [1,3,4,5]] for i in 1:6]
    irregular = IrregularFunctionalData(times,[[sin(i+t)+i*t+t^2 for t in ts]
        for (i,ts) in enumerate(times)])
    unique_times = [[.1,.3,.6,.9] .+ i/1000 for i in 1:6]
    untied = IrregularFunctionalData(unique_times,
        [[sin(i+t)+i*t+t^2 for t in ts] for (i,ts) in enumerate(unique_times)])
    mean = GridFunctionalData(zeros(1,length(grid)),grid)
    kernel = KernelSmoothing.GaussianKernel()

    function check_moments(raw, grouped)
        coordinates = unique(collect(zip(raw.x,raw.y)))
        @test collect(zip(grouped.x,grouped.y)) == coordinates
        for q in eachindex(grouped.z)
            ix = findall(i -> (raw.x[i],raw.y[i]) == coordinates[q],eachindex(raw.z))
            w,z = raw.weights[ix],raw.z[ix]
            @test grouped.weights[q] ≈ sum(w) rtol=1e-14
            @test grouped.z[q] ≈ dot(w,z)/sum(w) rtol=1e-13 atol=1e-14
            @test grouped.scatter[q] ≈ sum(w .* abs2.(z .- grouped.z[q])) + sum(raw.scatter[ix]) rtol=1e-12 atol=1e-13
        end
        prediction(x,y) = sin(x)+cos(y)
        raw_loss = sum(raw.weights .* abs2.(raw.z-prediction.(raw.x,raw.y)) + raw.scatter)
        grouped_loss = sum(grouped.weights .* abs2.(grouped.z-prediction.(grouped.x,grouped.y)) + grouped.scatter)
        @test grouped_loss ≈ raw_loss rtol=1e-13
    end

    # Independent raw-pair oracle: split complete subjects and query every
    # held-out coordinate, without combining either side of a fold.
    function raw_cv(pool,selection,kernel)
        errors = zeros(length(selection.bandwidths))
        for fold in unique(selection.assignments)
            held = selection.assignments[pool.subjects] .== fold
            train = .!held
            for (j,h) in enumerate(selection.bandwidths)
                p = KernelSmoothing.local_polynomial((pool.x[train],pool.y[train]),pool.z[train],
                    hcat(pool.x[held],pool.y[held]);bandwidth=h,kernel,weights=pool.weights[train])
                errors[j] += sum(pool.weights[held] .* abs2.(pool.z[held]-p) + pool.scatter[held])
            end
        end
        errors/sum(pool.weights)
    end

    for fd in (complete,masked,irregular,untied), weighting in (:obs,:subj), binning in (nothing,4)
        raw = FDA._covariance_pool(fd,mean,weighting,binning)
        collapsed = FDA._covariance_collapse_pairs(raw)
        grouped = FDA._covariance_pool(fd,mean,weighting,binning;aggregate=true)
        check_moments(raw,grouped)
        @test grouped.x == collapsed.x && grouped.y == collapsed.y
        @test grouped.z ≈ collapsed.z rtol=1e-13
        @test grouped.scatter ≈ collapsed.scatter rtol=1e-12 atol=1e-13
        if fd === untied && binning === nothing
            @test grouped.z == raw.z
            @test grouped.weights == raw.weights
        else
            @test length(grouped.z) < length(raw.z)
        end
        for name in (:times,:original_times,:squares,:diagonal_weights,:diagonal_subjects)
            @test getproperty(grouped,name) == getproperty(raw,name)
        end
        fit = StatsAPI.fit(SmoothedCovariance,fd,mean;grid,bandwidth=2.,kernel,weighting,binning)
        surface = FDA._local_linear_surface(raw,grid,grid,2.,kernel)
        @test fit.surface ≈ (surface+surface')/2 rtol=1e-12 atol=1e-13
        points = fit.noise_details.integration_grid
        fd === untied && @test points == grid[2:4]
        @test fit.noise_details.latent_diagonal ≈ FDA._rotated_covariance_diagonal(raw,points,2.,kernel) rtol=1e-12 atol=1e-13
        selected = StatsAPI.fit(SmoothedCovariance,fd,mean;grid,
            bandwidth=CurveCV([1.2,2.];folds=3,rng=MersenneTwister(71)),kernel,weighting,binning)
        @test selected.selection.errors ≈ raw_cv(raw,selected.selection,kernel) rtol=1e-12
        @test_throws ArgumentError StatsAPI.fit(SmoothedCovariance,fd,mean;
            grid,bandwidth=.001,kernel,weighting,binning)
    end

    # Cross-node Cartesian products, unequal grids and equal-time exclusions.
    grid2 = [0.,.35,.65,1.]
    second = GridFunctionalData([cos(i+t)+i*t^2 for i in 1:6,t in grid2],grid2)
    mean2 = GridFunctionalData(zeros(1,length(grid2)),grid2)
    for fd in (complete,irregular,untied), weighting in (:obs,:subj), binning in (nothing,4), exclude in (false,true)
        raw = FDA._cross_pool(fd,second,mean,mean2,weighting,exclude,binning)
        grouped = FDA._cross_pool(fd,second,mean,mean2,weighting,exclude,binning;aggregate=true)
        check_moments(raw,grouped)
        fit = StatsAPI.fit(SmoothedCrossCovariance,fd,second,mean,mean2;grid1=grid,grid2,
            bandwidth=(1.2,2.),kernel,weighting,binning,exclude_same_time=exclude)
        @test fit.surface ≈ FDA._local_linear_surface(raw,grid,grid2,(1.2,2.),kernel) rtol=1e-12 atol=1e-13
        selected = StatsAPI.fit(SmoothedCrossCovariance,fd,second,mean,mean2;grid1=grid,grid2,
            bandwidth=CurveCV([1.2,2.];folds=3,rng=MersenneTwister(71)),kernel,weighting,binning,exclude_same_time=exclude)
        @test selected.selection.errors ≈ raw_cv(raw,selected.selection,kernel) rtol=1e-12
    end
end
