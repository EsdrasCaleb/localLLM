package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_spRatingAbove_33_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testSpRatingAbove() throws NoSuchFieldException, IllegalAccessException {
        // Test setting a non-null value
        scannerSubscription.spRatingAbove("AAA");
        Field field = ScannerSubscription.class.getDeclaredField("m_spRatingAbove");
        field.setAccessible(true);
        assertEquals("AAA", field.get(scannerSubscription));
        // Test setting a null value
        scannerSubscription.spRatingAbove(null);
        assertNull(field.get(scannerSubscription));
    }
}
