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

public class EWrapperMsgGenerator_tickOptionComputation_2_3_Test {

    @Test
    void testTickOptionComputation() {
        // Arrange
        int tickerId = 123;
        int field = 2;
        double impliedVol = 0.2;
        double delta = 0.01;
        double modelPrice = 100.0;
        double pvDividend = 5.0;
        // Act
        String result = EWrapperMsgGenerator.tickOptionComputation(tickerId, field, impliedVol, delta, modelPrice, pvDividend);
        // Assert
        String expected = "id=" + tickerId + " " + TickType.getField(field) + ": vol = " + ((impliedVol >= 0 && impliedVol != Double.MAX_VALUE) ? Double.toString(impliedVol) : "N/A") + " delta = " + ((Math.abs(delta) <= 1) ? Double.toString(delta) : "N/A");
        if (field == TickType.MODEL_OPTION) {
            expected += ": modelPrice = " + ((modelPrice >= 0 && modelPrice != Double.MAX_VALUE) ? Double.toString(modelPrice) : "N/A") + ": pvDividend = " + ((pvDividend >= 0 && pvDividend != Double.MAX_VALUE) ? Double.toString(pvDividend) : "N/A");
        }
        assertEquals(expected, result);
    }
}
