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

public class EWrapperMsgGenerator_updateAccountValue_9_0_Test {

    @Test
    public void testUpdateAccountValue() {
        // Test data
        String key = "CashBalance";
        String value = "1000.50";
        String currency = "USD";
        String accountName = "Primary";
        // Expected result
        String expectedResult = "updateAccountValue: CashBalance 1000.50 USD Primary";
        // Actual result
        String actualResult = EWrapperMsgGenerator.updateAccountValue(key, value, currency, accountName);
        // Assertion
        assertEquals(expectedResult, actualResult, "The updateAccountValue method did not return the expected formatted string.");
    }
}
