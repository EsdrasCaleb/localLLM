package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class Platforms_getPlatform_3_0_Test {

    @InjectMocks
    private Platforms platforms;

    @Mock
    private ArrayList<String> platform;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
        platforms.setPlatform(new String[] { "Platform1", "Platform2", "Platform3" });
    }

    @Test
    void testGetPlatformValidIndex() {
        String result = platforms.getPlatform(1);
        assertEquals("Platform2", result);
    }

    @Test
    void testGetPlatformInvalidIndex() {
        String result = platforms.getPlatform(5);
        assertNull(result);
    }

    @Test
    void testGetPlatformBoundaryIndex() {
        String result = platforms.getPlatform(2);
        assertEquals("Platform3", result);
    }

    @Test
    void testGetPlatformNegativeIndex() {
        String result = platforms.getPlatform(-1);
        assertNull(result);
    }

    @Test
    void testGetPlatformEmptyPlatform() {
        platforms.setPlatform(new String[] {});
        String result = platforms.getPlatform(0);
        assertNull(result);
    }
}
