package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_excludeConvertible_39_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    void testExcludeConvertible() {
        scannerSubscription.excludeConvertible("YES");
        assertEquals("YES", scannerSubscription.excludeConvertible());
        scannerSubscription.excludeConvertible("NO");
        assertEquals("NO", scannerSubscription.excludeConvertible());
        scannerSubscription.excludeConvertible(null);
        assertNull(scannerSubscription.excludeConvertible());
    }
}
