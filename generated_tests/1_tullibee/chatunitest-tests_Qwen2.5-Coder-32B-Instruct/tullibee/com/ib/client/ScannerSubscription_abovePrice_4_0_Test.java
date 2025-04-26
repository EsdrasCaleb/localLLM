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
        scannerSubscription = Mockito.spy(new ScannerSubscription());
    }

    @Test
    public void testAbovePrice_DefaultValue() {
        // Given: A new ScannerSubscription object with default values
        // When: abovePrice() is called
        double result = scannerSubscription.abovePrice();
        // Then: It should return the default value of m_abovePrice
        assertEquals(Double.MAX_VALUE, result);
    }

    @Test
    public void testAbovePrice_SetValue() {
        // Given: A new ScannerSubscription object
        // When: abovePrice(double price) is called with a specific value
        double testValue = 150.75;
        scannerSubscription.abovePrice(testValue);
        // And: abovePrice() is called
        double result = scannerSubscription.abovePrice();
        // Then: It should return the set value of m_abovePrice
        assertEquals(testValue, result);
    }
}
