package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_numberOfRows_0_0_Test {

    private ScannerSubscription subscription;

    @BeforeEach
    public void setUp() {
        subscription = new ScannerSubscription();
    }

    @Test
    public void testNumberOfRows() {
        subscription.numberOfRows(10);
        assertEquals(10, subscription.numberOfRows());
        subscription.numberOfRows(20);
        assertEquals(20, subscription.numberOfRows());
        subscription.numberOfRows(ScannerSubscription.NO_ROW_NUMBER_SPECIFIED);
        assertEquals(ScannerSubscription.NO_ROW_NUMBER_SPECIFIED, subscription.numberOfRows());
    }
}
