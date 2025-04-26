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

class EWrapperMsgGenerator_tickPrice_0_1_Test {

    private EWrapperMsgGenerator eWrapperMsgGenerator;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.initMocks(this);
        eWrapperMsgGenerator = new EWrapperMsgGenerator();
    }

    @Test
    void testTickPrice() {
        // Arrange
        int tickerId = 123;
        int field = 456;
        double price = 789.0;
        int canAutoExecute = 1;
        // Act
        String result = eWrapperMsgGenerator.tickPrice(tickerId, field, price, canAutoExecute);
        // Assert
        assertEquals("id=123  field=456  price=789.0  noAutoExecute", result);
    }
}
