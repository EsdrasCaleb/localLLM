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

@ExtendWith(MockitoExtension.class)
public class EWrapperMsgGenerator_updateAccountValue_9_0_Test {

    // Test class
    @Test
    void testUpdateAccountValue() {
        // Arrange
        final String key = "key";
        final String value = "value";
        final String currency = "currency";
        final String accountName = "accountName";
        final String expectedResult = "updateAccountValue: " + key + " " + value + " " + currency + " " + accountName;
        // Act
        final String actualResult = EWrapperMsgGenerator.updateAccountValue(key, value, currency, accountName);
        // Assert
        assertEquals(expectedResult, actualResult);
    }
}
