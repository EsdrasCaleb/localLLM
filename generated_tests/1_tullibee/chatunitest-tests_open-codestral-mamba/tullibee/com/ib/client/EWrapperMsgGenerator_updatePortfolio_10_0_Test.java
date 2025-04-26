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

public class EWrapperMsgGenerator_updatePortfolio_10_0_Test {

    @Test
    public void testUpdatePortfolio() {
        Contract contract = Mockito.mock(Contract.class);
        int position = 10;
        double marketPrice = 150.50;
        double marketValue = 1505.00;
        double averageCost = 120.00;
        double unrealizedPNL = 30.50;
        double realizedPNL = 10.25;
        String accountName = "Test Account";
        String expectedMsg = "updatePortfolio: " + contract.toString() + " " + position + " " + marketPrice + " " + marketValue + " " + averageCost + " " + unrealizedPNL + " " + realizedPNL + " " + accountName;
        String actualMsg = EWrapperMsgGenerator.updatePortfolio(contract, position, marketPrice, marketValue, averageCost, unrealizedPNL, realizedPNL, accountName);
        assertEquals(expectedMsg, actualMsg);
    }
}
