package net.kencochrane.a4j.beans;

import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Platforms_getPlatform_3_0_Test {

    private Platforms platforms;

    @BeforeEach
    public void setUp() {
        platforms = new Platforms();
        platforms.setPlatform(new String[] { "Platform1", "Platform2", "Platform3" });
    }

    @Test
    public void testGetPlatform_InvalidIndex_Negative() {
        String result = platforms.getPlatform(-1);
        assertNull(result);
    }
}
