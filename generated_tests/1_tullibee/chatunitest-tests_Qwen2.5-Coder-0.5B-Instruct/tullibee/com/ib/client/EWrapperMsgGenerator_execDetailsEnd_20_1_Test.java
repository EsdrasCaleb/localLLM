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

class EWrapperMsgGenerator_execDetailsEnd_20_1_Test {

    @Test
    void execDetailsEnd() {
        // Arrange
        // Example request ID
        int reqId = 123;
        EWrapperMsgGenerator wrapperMsgGenerator = new EWrapperMsgGenerator();
        // Act
        String result = wrapperMsgGenerator.execDetailsEnd(reqId);
        // Assert
        assertEquals("123", result);
    }
}
