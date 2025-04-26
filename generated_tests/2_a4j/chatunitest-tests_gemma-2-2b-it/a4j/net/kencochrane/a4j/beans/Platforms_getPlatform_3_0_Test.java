package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Platforms_getPlatform_3_0_Test {

    @Test
    void testGetPlatform() {
        Platforms platforms = new Platforms();
        platforms.setPlatform(new String[] { "Windows", "Android", "iOS" });
        assertEquals("Windows", platforms.getPlatform(0));
        assertEquals("Android", platforms.getPlatform(1));
        assertEquals("iOS", platforms.getPlatform(2));
    }
}
