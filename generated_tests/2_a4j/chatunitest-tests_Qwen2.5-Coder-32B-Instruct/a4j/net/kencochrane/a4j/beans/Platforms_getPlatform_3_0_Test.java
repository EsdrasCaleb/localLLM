package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class Platforms_getPlatform_3_0_Test {

    @InjectMocks
    private Platforms platforms;

    @BeforeEach
    public void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        // Initialize the private ArrayList field using reflection
        Field platformField = Platforms.class.getDeclaredField("platform");
        platformField.setAccessible(true);
        platformField.set(platforms, new ArrayList<>());
    }

    @Test
    public void testGetPlatform_negativeIndex() throws Exception {
        ArrayList<String> platformList = new ArrayList<>();
        platformList.add("Platform1");
        Field platformField = Platforms.class.getDeclaredField("platform");
        platformField.setAccessible(true);
        platformField.set(platforms, platformList);
        // Act
        String result = platforms.getPlatform(-1);
        // Assert
        assertNull(result);
    }
}
