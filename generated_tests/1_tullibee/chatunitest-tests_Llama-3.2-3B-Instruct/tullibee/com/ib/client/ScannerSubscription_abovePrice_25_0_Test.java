package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_abovePrice_25_0_Test {

    @Test
    public void testAbovePrice_setNewPrice() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.abovePrice(10.0);
        assertEquals(10.0, scannerSubscription.abovePrice(), 0.01);
    }

    @Test
    public void testAbovePrice_noChange() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.abovePrice(Double.MAX_VALUE);
        assertEquals(Double.MAX_VALUE, scannerSubscription.abovePrice(), 0.01);
    }

    @Test
    public void testAbovePrice_negativeNumber() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        assertThrows(NumberFormatException.class, () -> scannerSubscription.abovePrice(-10.0));
    }

    @Test
    public void testAbovePrice_zero() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.abovePrice(0.0);
        assertEquals(0.0, scannerSubscription.abovePrice(), 0.01);
    }
}
