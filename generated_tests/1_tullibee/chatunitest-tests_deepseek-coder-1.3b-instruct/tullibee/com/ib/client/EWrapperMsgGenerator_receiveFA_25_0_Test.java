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

public class EWrapperMsgGenerator_receiveFA_25_0_Test {

    @Test
    public void testReceiveFA() {
        // Arrange
        int faDataType = 1;
        String xml = "test xml";
        String expected = EWrapperMsgGenerator.FINANCIAL_ADVISOR + " " + EClientSocket.faMsgTypeName(faDataType) + " " + xml;
        // Act
        String result = EWrapperMsgGenerator.receiveFA(faDataType, xml);
        // Assert
        assertEquals(expected, result);
    }
}
