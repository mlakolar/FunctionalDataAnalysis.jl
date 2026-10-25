import StatsAPI

function _test_score_prediction(result, legacy, projected, training)
    expected = legacy isa NamedTuple ? legacy.scores : legacy
    covariance = legacy isa NamedTuple ? legacy.covariances : nothing
    @test result isa ScorePrediction
    @test isconcretetype(typeof(result))
    @test result.scores == projected == expected
    @test result.covariances == covariance
    @test result.scores !== projected
    @test result.scores !== training
    if covariance !== nothing
        @test result.covariances !== covariance
        @test size(result.covariances) == (size(projected,2), size(projected,2), size(projected,1))
    end
    saved = deepcopy(training)
    result.scores[1,1] += 1
    @test training == saved
    @test result.scores != projected
end

function _test_prediction_contract(model, data)
    local legacy, projected, result
    @test_deprecated r"project_scores.*predict_scores" (legacy = StatsAPI.predict(model, data))
    @test_nowarn projected = project_scores(model, data)
    @test_nowarn result = predict_scores(model, data)
    if model isa MultivariateFPCA
        @test length(result) == length(projected) == length(legacy) == length(scores(model))
        for j in eachindex(result)
            _test_score_prediction(result[j], legacy[j], projected[j], scores(model)[j])
        end
    else
        _test_score_prediction(result, legacy, projected, scores(model))
    end
    fitted = StatsAPI.fitted(model)
    residual = StatsAPI.residuals(model)
    restored = reconstruct(model)
    if data isa MultivariateFunctionalData
        @test residual isa MultivariateFunctionalData
        @test ncomponents(residual) == ncomponents(data)
        for j in 1:ncomponents(data)
            @test component(fitted,j).values == component(restored,j).values
            _test_residual_layout(component(data,j), component(residual,j), component(fitted,j))
        end
    elseif data isa BasisFunctionalData
        @test residual isa BasisFunctionalData
        @test fitted.values == restored.values
        @test residual.basis == data.basis
        @test ids(residual) == ids(data)
        @test residual.coefs !== model.training_data.coefs
        @test StatsAPI.coef(residual) === residual.coefs
        @test StatsAPI.coef(data) === data.coefs
        points = collect(range(domain(data)...; length=31))
        latent = evaluate(model.mean,points) .+ scores(model)*transpose(basis_matrix(eigenfunctions(model),points))
        @test evaluate(residual,points) + latent ≈ evaluate(data,points) atol=1e-12
        @test !applicable(StatsAPI.coef, model)
    else
        @test fitted.values == restored.values
        _test_residual_layout(data, residual, fitted)
    end
end

function _test_residual_layout(data::GridFunctionalData, residual, fitted)
    @test residual isa GridFunctionalData
    @test residual.grid == data.grid
    @test residual.observed == data.observed
    @test ids(residual) == ids(data)
    @test residual.values !== data.values
    latent = evaluate(fitted, data.grid, PiecewiseLinear())
    observed = data.observed === nothing ? trues(size(data.values)) : data.observed
    @test (residual.values + latent)[observed] ≈ data.values[observed] atol=1e-12
    @test all(iszero, residual.values[.!observed])
end

function _test_residual_layout(data::IrregularFunctionalData, residual, fitted)
    @test residual isa IrregularFunctionalData
    @test residual.times == data.times
    @test residual.offsets == data.offsets
    @test domain(residual) == domain(data)
    @test ids(residual) == ids(data)
    @test residual.values !== data.values
    for i in 1:nobs(data)
        times, values = data[i]
        @test residual[i][2] + vec(evaluate(fitted[i:i], times, PiecewiseLinear())) ≈ values atol=1e-12
    end
end
