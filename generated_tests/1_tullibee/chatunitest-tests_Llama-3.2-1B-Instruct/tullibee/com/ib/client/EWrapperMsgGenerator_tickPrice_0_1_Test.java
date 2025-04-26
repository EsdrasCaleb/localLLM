package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

@ExtendWith(MockitoExtension.class)
public class EWrapperMsgGenerator_tickPrice_0_1_Test {

    @Mock
    private EWrapperMsgGenerator focal;

    @InjectMocks
    private EWrapperMsgGenerator instance;

    @Test
    public void testTickPrice() {
        // Arrange
        int tickerId = 1;
        int field = 1;
        double price = 100.0;
        int canAutoExecute = 1;
        // Act
        String result = instance.tickPrice(tickerId, field, price, canAutoExecute);
        // Assert
        assertEquals("id=1  field=1=100  canAutoExecute=1", result);
    }
}
