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

public class Platforms_getPlatform_3_1_Test {

    private Platforms platforms;

    @BeforeEach
    void setUp() {
        platforms = new Platforms();
    }

    @Test
    void testGetPlatformNullList() {
        try {
            Field platformField = Platforms.class.getDeclaredField("platform");
            platformField.setAccessible(true);
            platformField.set(platforms, null);
            assertThrows(NullPointerException.class, () -> platforms.getPlatform(0));
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Exception during reflection: " + e.getMessage());
        }
    }
}
