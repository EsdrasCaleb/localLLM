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
    void testUpdateAccountValue_ValidInput() {
        String key = "accountNumber";
        String value = "1000";
        String currency = "USD";
        String accountName = "Checking";
        String expectedOutput = "updateAccountValue: accountNumber 1000 USD Checking";
        String actualOutput = EWrapperMsgGenerator.updateAccountValue(key, value, currency, accountName);
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    void testUpdateAccountValue_NullKey() {
        String key = null;
        String value = "1000";
        String currency = "USD";
        String accountName = "Checking";
        String expectedOutput = "updateAccountValue: null 1000 USD Checking";
        String actualOutput = EWrapperMsgGenerator.updateAccountValue(key, value, currency, accountName);
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    void testUpdateAccountValue_EmptyKey() {
        String key = "";
        String value = "1000";
        String currency = "USD";
        String accountName = "Checking";
        String expectedOutput = "updateAccountValue:  1000 USD Checking";
        String actualOutput = EWrapperMsgGenerator.updateAccountValue(key, value, currency, accountName);
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    void testUpdateAccountValue_NullValue() {
        String key = "accountNumber";
        String value = null;
        String currency = "USD";
        String accountName = "Checking";
        String expectedOutput = "updateAccountValue: accountNumber null USD Checking";
        String actualOutput = EWrapperMsgGenerator.updateAccountValue(key, value, currency, accountName);
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    void testUpdateAccountValue_NullCurrency() {
        String key = "accountNumber";
        String value = "1000";
        String currency = null;
        String accountName = "Checking";
        String expectedOutput = "updateAccountValue: accountNumber 1000 null Checking";
        String actualOutput = EWrapperMsgGenerator.updateAccountValue(key, value, currency, accountName);
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    void testUpdateAccountValue_NullAccountName() {
        String key = "accountNumber";
        String value = "1000";
        String currency = "USD";
        String accountName = null;
        String expectedOutput = "updateAccountValue: accountNumber 1000 USD null";
        String actualOutput = EWrapperMsgGenerator.updateAccountValue(key, value, currency, accountName);
        assertEquals(expectedOutput, actualOutput);
    }
}
