package com.ib.client;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

class EWrapperMsgGenerator_updatePortfolio_10_1_Test {

    @Test
    void testUpdatePortfolio() throws NoSuchMethodException, IllegalAccessException, InvocationTargetException {
        // Test case 1: Normal input
        Contract contract = Mockito.mock(Contract.class);
        // Simulate contract details
        Mockito.when(contract.toString()).thenReturn("MockContract");
        String expectedMsg = "updatePortfolio: MockContract10 100.0 1000.0 50.0 20.0 10.0 Account1";
        String actualMsg = EWrapperMsgGenerator.updatePortfolio(contract, 10, 100.0, 1000.0, 50.0, 20.0, 10.0, "Account1");
        assertEquals(expectedMsg, actualMsg);
        // Test case 2: Zero position
        expectedMsg = "updatePortfolio: MockContract0 0.0 0.0 0.0 0.0 0.0 0.0 Account1";
        actualMsg = EWrapperMsgGenerator.updatePortfolio(contract, 0, 0.0, 0.0, 0.0, 0.0, 0.0, "Account1");
        assertEquals(expectedMsg, actualMsg);
        // Test case 3: Negative values
        expectedMsg = "updatePortfolio: MockContract-10 -100.0 -1000.0 -50.0 -20.0 -10.0 Account1";
        actualMsg = EWrapperMsgGenerator.updatePortfolio(contract, -10, -100.0, -1000.0, -50.0, -20.0, -10.0, "Account1");
        assertEquals(expectedMsg, actualMsg);
        // Test case 4: Null contract (should handle gracefully)
        expectedMsg = "updatePortfolio: null0 0.0 0.0 0.0 0.0 0.0 0.0 Account1";
        actualMsg = EWrapperMsgGenerator.updatePortfolio(null, 0, 0.0, 0.0, 0.0, 0.0, 0.0, "Account1");
        assertEquals(expectedMsg, actualMsg);
        // Test case 5: Empty account name
        expectedMsg = "updatePortfolio: MockContract10 100.0 1000.0 50.0 20.0 10.0 ";
        actualMsg = EWrapperMsgGenerator.updatePortfolio(contract, 10, 100.0, 1000.0, 50.0, 20.0, 10.0, "");
        assertEquals(expectedMsg, actualMsg);
    }
}

// Dummy class for testing
class Contract {

    @Override
    public String toString() {
        return "MockContract";
    }
}
