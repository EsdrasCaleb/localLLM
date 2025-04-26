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

class EWrapperMsgGenerator_nextValidId_13_1_Test {

    @Test
    void testNextValidId() {
        // Create a mock instance of EWrapperMsgGenerator
        EWrapperMsgGenerator eWrapperMsgGeneratorMock = Mockito.mock(EWrapperMsgGenerator.class);
        // Call the nextValidId method with a sample orderId
        String result = eWrapperMsgGeneratorMock.nextValidId(12345);
        // Verify the result
        Mockito.verify(eWrapperMsgGeneratorMock).nextValidId(12345);
        assertEquals("SCANNER_PARAMETERS: FA: 12345", result);
    }
}
