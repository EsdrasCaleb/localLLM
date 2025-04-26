package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_belowPrice_26_4_Test {

    @Test
    public void testBelowPrice() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.belowPrice(10.0);
        assertEquals(10.0, scannerSubscription.belowPrice(), 0.01);
    }

    @Test
    public void testBelowPriceWithNegativeNumber() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        assertThrows(NullPointerException.class, () -> scannerSubscription.belowPrice(-10.0));
    }

    @Test
    public void testBelowPriceWithZero() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.belowPrice(0.0);
        assertEquals(0.0, scannerSubscription.belowPrice(), 0.01);
    }

    @Test
    public void testBelowPriceWithMaxDouble() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.belowPrice(Double.MAX_VALUE);
        assertEquals(Double.MAX_VALUE, scannerSubscription.belowPrice(), 0.01);
    }

    @Test
    public void testBelowPriceWithMinDouble() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.belowPrice(Double.MIN_VALUE);
        assertEquals(Double.MIN_VALUE, scannerSubscription.belowPrice(), 0.01);
    }
}
