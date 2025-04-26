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

public class EWrapperMsgGenerator_tickPrice_0_0_Test {

    @Test
    public void testTickPrice_CanAutoExecute() {
        int tickerId = 1;
        // Assuming 0 corresponds to a valid TickType
        int field = 0;
        double price = 100.50;
        // Indicates auto-execution is allowed
        int canAutoExecute = 1;
        String expected = "id=1  " + TickType.getField(field) + "=100.5 canAutoExecute";
        String actual = EWrapperMsgGenerator.tickPrice(tickerId, field, price, canAutoExecute);
        assertEquals(expected, actual);
    }

    @Test
    public void testTickPrice_NoAutoExecute() {
        int tickerId = 2;
        // Assuming 1 corresponds to a valid TickType
        int field = 1;
        double price = 200.75;
        // Indicates auto-execution is not allowed
        int canAutoExecute = 0;
        String expected = "id=2  " + TickType.getField(field) + "=200.75 noAutoExecute";
        String actual = EWrapperMsgGenerator.tickPrice(tickerId, field, price, canAutoExecute);
        assertEquals(expected, actual);
    }

    @Test
    public void testTickPrice_NegativePrice() {
        int tickerId = 3;
        // Assuming 2 corresponds to a valid TickType
        int field = 2;
        // Testing with negative price
        double price = -50.0;
        // Indicates auto-execution is allowed
        int canAutoExecute = 1;
        String expected = "id=3  " + TickType.getField(field) + "=-50.0 canAutoExecute";
        String actual = EWrapperMsgGenerator.tickPrice(tickerId, field, price, canAutoExecute);
        assertEquals(expected, actual);
    }

    @Test
    public void testTickPrice_ZeroPrice() {
        int tickerId = 4;
        // Assuming 3 corresponds to a valid TickType
        int field = 3;
        // Testing with zero price
        double price = 0.0;
        // Indicates auto-execution is not allowed
        int canAutoExecute = 0;
        String expected = "id=4  " + TickType.getField(field) + "=0.0 noAutoExecute";
        String actual = EWrapperMsgGenerator.tickPrice(tickerId, field, price, canAutoExecute);
        assertEquals(expected, actual);
    }
}
