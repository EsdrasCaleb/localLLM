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

class EWrapperMsgGenerator_updateAccountValue_9_0_Test {

    @Test
    void testUpdateAccountValue() {
        // Arrange
        String key = "key";
        String value = "value";
        String currency = "USD";
        String accountName = "accountName";
        String expectedResult = "updateAccountValue: " + key + " " + value + " " + currency + " " + accountName;
        // Act
        String result = EWrapperMsgGenerator.updateAccountValue(key, value, currency, accountName);
        // Assert
        assertEquals(expectedResult, result);
    }
}
