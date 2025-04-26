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

public class EWrapperMsgGenerator_updateAccountValue_9_1_Test {

    @Test
    void testUpdateAccountValue() {
        EWrapperMsgGenerator eWrapperMsgGenerator = Mockito.mock(EWrapperMsgGenerator.class);
        String key = "key";
        String value = "value";
        String currency = "EUR";
        String accountName = "accountName";
        // Call the method under test
        String result = eWrapperMsgGenerator.updateAccountValue(key, value, currency, accountName);
        // Assert the result
        assertEquals("updateAccountValue: key value EUR accountName", result);
    }
}
