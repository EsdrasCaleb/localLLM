package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_abovePrice_4_0_Test {

    @Spy
    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = Mockito.spy(new ScannerSubscription());
    }

    @Test
    public void testAbovePrice() {
        double expected = 100.0;
        scannerSubscription.abovePrice(expected);
        double actual = scannerSubscription.abovePrice();
        assertEquals(expected, actual);
    }
}
