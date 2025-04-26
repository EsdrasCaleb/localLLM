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

public class EWrapperMsgGenerator_tickPrice_0_1_Test {

    @Test
    public void testTickPrice() {
        // Arrange
        int tickerId = 123;
        int field = 1;
        double price = 123.45;
        int canAutoExecute = 1;
        // Act
        String result = EWrapperMsgGenerator.tickPrice(tickerId, field, price, canAutoExecute);
        // Assert
        assertEquals("id=123  FIELD=1=123.45 canAutoExecute", result);
    }
}
