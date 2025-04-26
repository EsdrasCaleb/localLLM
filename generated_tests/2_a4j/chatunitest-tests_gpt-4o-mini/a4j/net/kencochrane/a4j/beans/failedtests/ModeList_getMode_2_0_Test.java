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

class ModeList_getMode_2_0_Test {

    private ModeList modeList;

    @BeforeEach
    void setUp() {
        modeList = new ModeList();
    }

    @Test
    void testGetMode_WhenModesListIsEmpty_ReturnsNull() {
        assertNull(modeList.getMode("AnyMode"));
    }

    @Test
    void testGetMode_WhenModesListIsNull_ReturnsNull() throws NoSuchFieldException, IllegalAccessException {
        Field modesField = ModeList.class.getDeclaredField("modes");
        modesField.setAccessible(true);
        modesField.set(modeList, null);
        assertNull(modeList.getMode("AnyMode"));
    }

    @Test
    void testGetMode_WhenModeExists_ReturnsMode() {
        Mode mode = new Mode("TestMode");
        addMode(mode);
        Mode result = modeList.getMode("TestMode");
        assertNotNull(result);
        assertEquals("TestMode", result.getModeName());
    }

    @Test
    void testGetMode_WhenModeDoesNotExist_ReturnsNull() {
        Mode mode = new Mode("TestMode");
        addMode(mode);
        assertNull(modeList.getMode("NonExistentMode"));
    }

    @Test
    void testGetMode_WhenModeNameIsCaseInsensitive_ReturnsMode() {
        Mode mode = new Mode("TestMode");
        addMode(mode);
        Mode result = modeList.getMode("testmode");
        assertNotNull(result);
        assertEquals("TestMode", result.getModeName());
    }

    private void addMode(Mode mode) {
        try {
            Field modesField = ModeList.class.getDeclaredField("modes");
            modesField.setAccessible(true);
            ArrayList<Mode> modes = (ArrayList<Mode>) modesField.get(modeList);
            if (modes == null) {
                modes = new ArrayList<>();
            }
            modes.add(mode);
            modesField.set(modeList, modes);
        } catch (Exception e) {
            fail("Failed to add mode: " + e.getMessage());
        }
    }
}

class Mode {

    private String modeName;

    public Mode(String modeName) {
        this.modeName = modeName;
    }

    public String getModeName() {
        return modeName;
    }
}
