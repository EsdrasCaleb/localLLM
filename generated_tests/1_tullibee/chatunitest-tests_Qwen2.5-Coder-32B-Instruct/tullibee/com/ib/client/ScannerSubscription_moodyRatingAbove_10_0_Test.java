package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_moodyRatingAbove_10_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testMoodyRatingAbove_DefaultValue() {
        // Default value should be null as it is not initialized in the constructor
        assertNull(scannerSubscription.moodyRatingAbove());
    }

    @Test
    public void testMoodyRatingAbove_SetValue() throws Exception {
        // Set the value using reflection
        Field moodyRatingAboveField = ScannerSubscription.class.getDeclaredField("m_moodyRatingAbove");
        moodyRatingAboveField.setAccessible(true);
        moodyRatingAboveField.set(scannerSubscription, "A1");
        // Verify the value returned by moodyRatingAbove()
        assertEquals("A1", scannerSubscription.moodyRatingAbove());
    }
}
