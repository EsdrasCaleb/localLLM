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
public class EWrapperMsgGenerator_tickSize_1_1_Test {

    @Mock
    private EWrapperMsgGenerator focal;

    @Test
    public void testTickSize() {
        // Arrange
        int tickerId = 1;
        int field = 1;
        int size = 10;
        // Act
        String result = focal.tickSize(tickerId, field, size);
        // Assert
        assertEquals("id=1  FINANCIAL_ADVISOR=10", result);
    }
}
