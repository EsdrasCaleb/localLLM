package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.lang.reflect.Field;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
class ModeList_getMode_2_0_Test {

    @InjectMocks
    private ModeList modeList;

    @BeforeEach
    void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        // Initialize the modes ArrayList using reflection
        Field field = ModeList.class.getDeclaredField("modes");
        field.setAccessible(true);
        field.set(modeList, new ArrayList<>());
    }

    @Test
    void testGetMode_ModeFound() {
        // Arrange
        Mode mockMode = new Mode();
        mockMode.setModeName("TestMode");
        modeList.getAllModes().add(mockMode);
        // Act
        net.kencochrane.a4j.beans.Mode result = modeList.getMode("TestMode");
        // Assert
        assertNotNull(result);
        assertEquals("TestMode", result.getModeName());
    }

    @Test
    void testGetMode_ModeNotFound() {
        // Arrange
        Mode mockMode = new Mode();
        mockMode.setModeName("AnotherMode");
        modeList.getAllModes().add(mockMode);
        // Act
        net.kencochrane.a4j.beans.Mode result = modeList.getMode("testmode");
        // Assert
        assertNull(result);
    }

    @Test
    void testGetMode_EmptyList() {
        // Arrange - modes list is already empty
        // Act
        net.kencochrane.a4j.beans.Mode result = modeList.getMode("testmode");
        // Assert
        assertNull(result);
    }

    @Test
    void testGetMode_NullList() throws Exception {
        // Arrange
        Field field = ModeList.class.getDeclaredField("modes");
        field.setAccessible(true);
        field.set(modeList, null);
        // Act
        net.kencochrane.a4j.beans.Mode result = modeList.getMode("testmode");
        // Assert
        assertNull(result);
    }

    // Helper class Mode
    static class Mode {

        private String modeName;

        public String getModeName() {
            return modeName;
        }

        public void setModeName(String modeName) {
            this.modeName = modeName;
        }
    }
}
