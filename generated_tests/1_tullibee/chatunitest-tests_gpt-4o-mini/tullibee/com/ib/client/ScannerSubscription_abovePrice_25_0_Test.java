package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_abovePrice_25_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testAbovePriceSetsValue() {
        double testPrice = 100.50;
        scannerSubscription.abovePrice(testPrice);
        assertEquals(testPrice, scannerSubscription.abovePrice());
    }

    @Test
    public void testAbovePriceSetsNegativeValue() {
        double testPrice = -50.25;
        scannerSubscription.abovePrice(testPrice);
        assertEquals(testPrice, scannerSubscription.abovePrice());
    }

    @Test
    public void testAbovePriceSetsZero() {
        double testPrice = 0.0;
        scannerSubscription.abovePrice(testPrice);
        assertEquals(testPrice, scannerSubscription.abovePrice());
    }

    @Test
    public void testAbovePriceSetsMaxValue() {
        double testPrice = Double.MAX_VALUE;
        scannerSubscription.abovePrice(testPrice);
        assertEquals(testPrice, scannerSubscription.abovePrice());
    }

    @Test
    public void testAbovePriceSetsMinValue() {
        double testPrice = Double.MIN_VALUE;
        scannerSubscription.abovePrice(testPrice);
        assertEquals(testPrice, scannerSubscription.abovePrice());
    }
}
