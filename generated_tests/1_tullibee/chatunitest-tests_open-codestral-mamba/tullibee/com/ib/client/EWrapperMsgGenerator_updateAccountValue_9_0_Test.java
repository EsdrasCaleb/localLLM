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
        String key = "key";
        String value = "value";
        String currency = "currency";
        String accountName = "accountName";
        String expected = "updateAccountValue: " + key + " " + value + " " + currency + " " + accountName;
        String actual = EWrapperMsgGenerator.updateAccountValue(key, value, currency, accountName);
        assertEquals(expected, actual);
    }
}
