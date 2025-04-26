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

public class EWrapperMsgGenerator_tickOptionComputation_2_2_Test {

    @Test
    public void testTickOptionComputation() {
        // Arrange
        int tickerId = 123;
        int field = 4;
        double impliedVol = 0.3;
        double delta = 0.4;
        double modelPrice = 0.5;
        double pvDividend = 0.6;
        String expected = "id=123  TICK: vol = 0.3 delta = 0.4: modelPrice = 0.5: pvDividend = 0.6";
        // Act
        String result = EWrapperMsgGenerator.tickOptionComputation(tickerId, field, impliedVol, delta, modelPrice, pvDividend);
        // Assert
        assertEquals(expected, result);
    }
}
