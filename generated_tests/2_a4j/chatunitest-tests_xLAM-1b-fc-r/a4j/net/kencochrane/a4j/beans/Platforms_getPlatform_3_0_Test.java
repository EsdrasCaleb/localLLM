package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class Platforms_getPlatform_3_0_Test {

    Platforms platforms;

    @BeforeEach
    void setUp() {
        platforms = new Platforms();
        platforms.setPlatform(new String[] { "Platform 1", "Platform 2", "Platform 3" });
    }

    @Test
    void testGetPlatform() {
        assertEquals("Platform 1", platforms.getPlatform(0));
        assertEquals("Platform 2", platforms.getPlatform(1));
        assertEquals("Platform 3", platforms.getPlatform(2));
    }
}
