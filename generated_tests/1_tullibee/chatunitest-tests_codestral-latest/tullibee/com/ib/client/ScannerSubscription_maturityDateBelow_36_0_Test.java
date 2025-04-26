package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_maturityDateBelow_36_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testMaturityDateBelow() throws NoSuchFieldException, IllegalAccessException {
        // Test setting a valid maturity date
        scannerSubscription.maturityDateBelow("2023-12-31");
        Field maturityDateBelowField = ScannerSubscription.class.getDeclaredField("m_maturityDateBelow");
        maturityDateBelowField.setAccessible(true);
        assertEquals("2023-12-31", maturityDateBelowField.get(scannerSubscription));
        // Test setting null maturity date
        scannerSubscription.maturityDateBelow(null);
        assertNull(maturityDateBelowField.get(scannerSubscription));
        // Test setting an empty maturity date
        scannerSubscription.maturityDateBelow("");
        assertEquals("", maturityDateBelowField.get(scannerSubscription));
    }
}
