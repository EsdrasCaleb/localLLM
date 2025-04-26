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

    @Test
    public void testGetPlatform() {
        // Create a mock object of Platforms
        Platforms platforms = Mockito.mock(Platforms.class);
        // Create an ArrayList to simulate the platform list
        ArrayList<String> mockPlatforms = new ArrayList<>();
        mockPlatforms.add("Windows");
        mockPlatforms.add("Mac");
        mockPlatforms.add("Linux");
        // Set the mock behavior of the getPlatform method
        Mockito.when(platforms.getPlatform(1)).thenReturn(mockPlatforms.get(1));
        Mockito.when(platforms.getPlatform(2)).thenReturn(mockPlatforms.get(2));
        Mockito.when(platforms.getPlatform(3)).thenReturn(null);
        // Call the method under test
        String platform1 = platforms.getPlatform(1);
        String platform2 = platforms.getPlatform(2);
        String platform3 = platforms.getPlatform(3);
        // Verify the results
        assertEquals("Mac", platform1);
        assertEquals("Linux", platform2);
        assertEquals(null, platform3);
    }
}
