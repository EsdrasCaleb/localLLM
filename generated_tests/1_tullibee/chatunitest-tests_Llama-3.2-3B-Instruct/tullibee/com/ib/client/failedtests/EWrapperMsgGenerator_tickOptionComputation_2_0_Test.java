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
public class EWrapperMsgGenerator_tickOptionComputation_2_0_Test {

    @Mock
    private TickType tickType;

    @Test
    public void testTickOptionComputationNormalCase() {
        // Arrange
        int tickerId = 1;
        int field = TickType.MODEL_OPTION;
        double impliedVol = 0.5;
        double delta = 0.3;
        double modelPrice = 10.2;
        double pvDividend = 5.1;
        when(tickType.getField(field)).thenReturn("FIELD_VALUE");
        // Act
        String result = EWrapperMsgGenerator.tickOptionComputation(tickerId, field, impliedVol, delta, modelPrice, pvDividend);
        // Assert
        assertEquals("id=1  FIELD_VALUE: vol = 0.5 delta = 0.3 modelPrice = 10.2 pvDividend = 5.1", result);
    }

    @Test
    public void testTickOptionComputationNullImpliedVol() {
        // Arrange
        int tickerId = 1;
        int field = TickType.MODEL_OPTION;
        double impliedVol = Double.MAX_VALUE;
        double delta = 0.3;
        double modelPrice = 10.2;
        double pvDividend = 5.1;
        when(tickType.getField(field)).thenReturn("FIELD_VALUE");
        // Act
        String result = EWrapperMsgGenerator.tickOptionComputation(tickerId, field, impliedVol, delta, modelPrice, pvDividend);
        // Assert
        assertEquals("id=1  FIELD_VALUE: vol = N/A delta = 0.3 modelPrice = 10.2 pvDividend = 5.1", result);
    }

    @Test
    public void testTickOptionComputationNullDelta() {
        // Arrange
        int tickerId = 1;
        int field = TickType.MODEL_OPTION;
        double impliedVol = 0.5;
        double delta = Double.MAX_VALUE;
        double modelPrice = 10.2;
        double pvDividend = 5.1;
        when(tickType.getField(field)).thenReturn("FIELD_VALUE");
        // Act
        String result = EWrapperMsgGenerator.tickOptionComputation(tickerId, field, impliedVol, delta, modelPrice, pvDividend);
        // Assert
        assertEquals("id=1  FIELD_VALUE: vol = 0.5 delta = N/A modelPrice = 10.2 pvDividend = 5.1", result);
    }

    @Test
    public void testTickOptionComputationNullModelPrice() {
        // Arrange
        int tickerId = 1;
        int field = TickType.MODEL_OPTION;
        double impliedVol = 0.5;
        double delta = 0.3;
        double modelPrice = Double.MAX_VALUE;
        double pvDividend = 5.1;
        when(tickType.getField(field)).thenReturn("FIELD_VALUE");
        // Act
        String result = EWrapperMsgGenerator.tickOptionComputation(tickerId, field, impliedVol, delta, modelPrice, pvDividend);
        // Assert
        assertEquals("id=1  FIELD_VALUE: vol = 0.5 delta = 0.3 modelPrice = N/A pvDividend = 5.1", result);
    }

    @Test
    public void testTickOptionComputationNullPvDividend() {
        // Arrange
        int tickerId = 1;
        int field = TickType.MODEL_OPTION;
        double impliedVol = 0.5;
        double delta = 0.3;
        double modelPrice = 10.2;
        double pvDividend = Double.MAX_VALUE;
        when(tickType.getField(field)).thenReturn("FIELD_VALUE");
        // Act
        String result = EWrapperMsgGenerator.tickOptionComputation(tickerId, field, impliedVol, delta, modelPrice, pvDividend);
        // Assert
        assertEquals("id=1  FIELD_VALUE: vol = 0.5 delta = 0.3 modelPrice = 10.2 pvDividend = N/A", result);
    }

    @Test
    public void testTickOptionComputationNormalCaseWithNullTickType() {
        // Arrange
        int tickerId = 1;
        // Invalid field value
        int field = 0;
        double impliedVol = 0.5;
        double delta = 0.3;
        double modelPrice = 10.2;
        double pvDividend = 5.1;
    }
}
