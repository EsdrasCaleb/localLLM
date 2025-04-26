package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_belowPrice_5_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testBelowPriceDefault() {
        // Test the default value of m_belowPrice
        assertEquals(Double.MAX_VALUE, scannerSubscription.belowPrice());
    }

    @Test
    public void testBelowPriceSetValue() {
        // Set a new value for m_belowPrice
        double testValue = 100.50;
        scannerSubscription.belowPrice(testValue);
        // Test if the value is set correctly
        assertEquals(testValue, scannerSubscription.belowPrice());
    }

    @Test
    public void testBelowPriceNegativeValue() {
        // Set a negative value for m_belowPrice
        double testValue = -50.75;
        scannerSubscription.belowPrice(testValue);
        // Test if the negative value is set correctly
        assertEquals(testValue, scannerSubscription.belowPrice());
    }

    @Test
    public void testBelowPriceZero() {
        // Set m_belowPrice to zero
        scannerSubscription.belowPrice(0.0);
        // Test if the value is set to zero correctly
        assertEquals(0.0, scannerSubscription.belowPrice());
    }
}
