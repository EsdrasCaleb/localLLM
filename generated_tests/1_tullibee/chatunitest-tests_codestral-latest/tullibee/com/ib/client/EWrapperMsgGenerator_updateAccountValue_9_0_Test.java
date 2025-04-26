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
        String key = "key1";
        String value = "value1";
        String currency = "USD";
        String accountName = "account1";
        String expected = "updateAccountValue: " + key + " " + value + " " + currency + " " + accountName;
        String result = EWrapperMsgGenerator.updateAccountValue(key, value, currency, accountName);
        assertEquals(expected, result);
    }
}
