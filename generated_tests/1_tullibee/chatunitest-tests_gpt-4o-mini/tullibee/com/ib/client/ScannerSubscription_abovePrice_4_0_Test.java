package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_abovePrice_4_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testAbovePriceDefault() {
        // Test the default value of m_abovePrice
        assertEquals(Double.MAX_VALUE, scannerSubscription.abovePrice());
    }

    @Test
    public void testAbovePriceSetValue() {
        // Set a specific value for m_abovePrice and test it
        double testPrice = 100.50;
        scannerSubscription.abovePrice(testPrice);
        assertEquals(testPrice, scannerSubscription.abovePrice());
    }

    @Test
    public void testAbovePriceNegativeValue() {
        // Set a negative value for m_abovePrice and test it
        double testPrice = -50.00;
        scannerSubscription.abovePrice(testPrice);
        assertEquals(testPrice, scannerSubscription.abovePrice());
    }

    @Test
    public void testAbovePriceZeroValue() {
        // Set zero as the value for m_abovePrice and test it
        double testPrice = 0.00;
        scannerSubscription.abovePrice(testPrice);
        assertEquals(testPrice, scannerSubscription.abovePrice());
    }
}
