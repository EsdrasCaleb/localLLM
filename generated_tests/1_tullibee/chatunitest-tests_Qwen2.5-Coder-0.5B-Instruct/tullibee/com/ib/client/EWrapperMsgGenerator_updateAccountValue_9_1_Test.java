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

class EWrapperMsgGenerator_updateAccountValue_9_1_Test {

    @Test
    void testUpdateAccountValue() {
        // Arrange
        String key = "exampleKey";
        String value = "exampleValue";
        String currency = "USD";
        String accountName = "exampleAccount";
        // Create an instance of the focal class
        EWrapperMsgGenerator wrapperMsgGenerator = new EWrapperMsgGenerator();
        // Invoke the method with the provided parameters
        String result = wrapperMsgGenerator.updateAccountValue(key, value, currency, accountName);
        // Assert the method's output matches the expected result
        assertEquals("updateAccountValue: exampleKey exampleValue USD exampleAccount", result);
    }
}
