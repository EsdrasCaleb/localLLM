package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

class EWrapperMsgGenerator_updatePortfolio_10_0_Test {

    @Test
    void updatePortfolioTest(MockedStatic<EWrapperMsgGenerator> mocked) {
        // Given
        Contract contract = new Contract();
        int position = 1;
        double marketPrice = 100.0;
        double marketValue = 200.0;
        double averageCost = 50.0;
        double unrealizedPNL = 10.0;
        double realizedPNL = 5.0;
        String accountName = "testAccount";
        // When
        String result = EWrapperMsgGenerator.updatePortfolio(contract, position, marketPrice, marketValue, averageCost, unrealizedPNL, realizedPNL, accountName);
        // Then
        String expected = "updatePortfolio: " + contractMsg(contract) + position + " " + marketPrice + " " + marketValue + " " + averageCost + " " + unrealizedPNL + " " + realizedPNL + " " + accountName;
        assertEquals(expected, result);
    }

    private String contractMsg(Contract contract) {
        // Mock the contract's methods to return specific values
        return "contractMsg";
    }
}
