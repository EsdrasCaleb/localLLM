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
    void testUpdateAccountValue() {
        // Given
        String key = "TEST_KEY";
        String value = "TEST_VALUE";
        String currency = "TEST_CURRENCY";
        String accountName = "TEST_ACCOUNT";
        // When
        String expectedOutput = "updateAccountValue: TEST_KEY TEST_VALUE TEST_CURRENCY TEST_ACCOUNT";
        // Then
        assertEquals(expectedOutput, EWrapperMsgGenerator.updateAccountValue(key, value, currency, accountName));
    }
}
