package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class Platforms_toString_4_0_Test {

    @Mock
    private Platforms platforms;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testToString() {
        ArrayList<String> platformList = new ArrayList<>();
        platformList.add("Platform1");
        platformList.add("Platform2");
        when(platforms.getPlatformsArray()).thenReturn(platformList);
        String expected = "# of Platforms = 2\n" + "Platform - Platform1\n" + "Platform - Platform2\n";
        assertEquals(expected, platforms.toString());
    }

    @Test
    public void testToStringEmptyList() {
        when(platforms.getPlatformsArray()).thenReturn(new ArrayList<>());
        String expected = "Platforms is null or size 0\n";
        assertEquals(expected, platforms.toString());
    }

    @Test
    public void testToStringNullList() {
        when(platforms.getPlatformsArray()).thenReturn(null);
        String expected = "Platforms is null or size 0\n";
        assertEquals(expected, platforms.toString());
    }
}
