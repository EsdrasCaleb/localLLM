package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class Platforms_getPlatform_3_0_Test {

    @Test
    void testGetPlatformNegativeIndex() {
        Platforms platforms = new Platforms();
        String[] platformsArray = { "Android", "iOS", "Web" };
        platforms.setPlatform(platformsArray);
        String platform = platforms.getPlatform(-1);
        assertNull(platform);
    }
}
