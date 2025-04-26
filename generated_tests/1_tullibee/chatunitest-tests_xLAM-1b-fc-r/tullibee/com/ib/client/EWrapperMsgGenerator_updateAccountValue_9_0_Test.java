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
        String key = "key";
        String value = "value";
        String currency = "currency";
        String accountName = "accountName";
        String expectedResult = "updateAccountValue: " + key + " " + value + " " + currency + " " + accountName;
        EWrapperMsgGenerator eWrapperMsgGenerator = new EWrapperMsgGenerator();
        String result = eWrapperMsgGenerator.updateAccountValue(key, value, currency, accountName);
        assertEquals(expectedResult, result);
    }
}
