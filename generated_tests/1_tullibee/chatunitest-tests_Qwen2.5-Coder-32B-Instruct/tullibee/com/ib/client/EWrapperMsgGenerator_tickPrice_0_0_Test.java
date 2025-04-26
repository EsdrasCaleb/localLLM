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
    public void testTickPriceWithAutoExecute() {
        int tickerId = 12345;
        // Assuming 1 corresponds to a valid TickType field
        int field = 1;
        double price = 150.75;
        // can auto-execute
        int canAutoExecute = 1;
        String expected = "id=12345  1=150.75 canAutoExecute";
        String result = EWrapperMsgGenerator.tickPrice(tickerId, field, price, canAutoExecute);
        assertEquals(expected, result);
    }

    @Test
    public void testTickPriceWithoutAutoExecute() {
        int tickerId = 67890;
        // Assuming 2 corresponds to a valid TickType field
        int field = 2;
        double price = 200.25;
        // cannot auto-execute
        int canAutoExecute = 0;
        String expected = "id=67890  2=200.25 noAutoExecute";
        String result = EWrapperMsgGenerator.tickPrice(tickerId, field, price, canAutoExecute);
        assertEquals(expected, result);
    }

    @Test
    public void testTickPriceWithZeroPrice() {
        int tickerId = 54321;
        // Assuming 3 corresponds to a valid TickType field
        int field = 3;
        double price = 0.0;
        // can auto-execute
        int canAutoExecute = 1;
        String expected = "id=54321  3=0.0 canAutoExecute";
        String result = EWrapperMsgGenerator.tickPrice(tickerId, field, price, canAutoExecute);
        assertEquals(expected, result);
    }

    @Test
    public void testTickPriceWithNegativePrice() {
        int tickerId = 98765;
        // Assuming 4 corresponds to a valid TickType field
        int field = 4;
        double price = -100.0;
        // cannot auto-execute
        int canAutoExecute = 0;
        String expected = "id=98765  4=-100.0 noAutoExecute";
        String result = EWrapperMsgGenerator.tickPrice(tickerId, field, price, canAutoExecute);
        assertEquals(expected, result);
    }

    @Test
    public void testTickPriceWithLargePrice() {
        int tickerId = 112233;
        // Assuming 5 corresponds to a valid TickType field
        int field = 5;
        double price = 9999999.99;
        // can auto-execute
        int canAutoExecute = 1;
        String expected = "id=112233  5=9999999.99 canAutoExecute";
        String result = EWrapperMsgGenerator.tickPrice(tickerId, field, price, canAutoExecute);
        assertEquals(expected, result);
    }
}
