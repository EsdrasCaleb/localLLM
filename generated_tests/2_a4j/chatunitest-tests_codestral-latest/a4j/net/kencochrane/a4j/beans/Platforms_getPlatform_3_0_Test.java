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
    void testGetPlatformNegativeIndex() {
        String result = platforms.getPlatform(-1);
        assertNull(result);
    }
}
