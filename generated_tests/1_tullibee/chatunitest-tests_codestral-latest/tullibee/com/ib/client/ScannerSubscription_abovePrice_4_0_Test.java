package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class ScannerSubscription_abovePrice_4_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testAbovePrice() {
        // Test default value
        assertEquals(Double.MAX_VALUE, scannerSubscription.abovePrice());
        // Test setting a new value
        double newAbovePrice = 100.50;
        scannerSubscription.abovePrice(newAbovePrice);
        assertEquals(newAbovePrice, scannerSubscription.abovePrice());
    }
}
