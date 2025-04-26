package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class ModeList_getMode_2_0_Test {

    @Test
    void testGetMode_emptyList() {
        ModeList modeList = new ModeList();
        // Initialize the list
        modeList.modes = new ArrayList<>();
        Mode foundMode = modeList.getMode("Mode1");
        assertNull(foundMode);
    }

    @Test
    void testGetMode_nullList() {
        ModeList modeList = new ModeList();
        modeList.modes = null;
        Mode foundMode = modeList.getMode("Mode1");
        assertNull(foundMode);
    }
}

// Mode class (needed for testing)
class Mode {

    private String modeName;

    public Mode(String modeName) {
        this.modeName = modeName;
    }

    // Default constructor
    public Mode() {
    }

    public String getModeName() {
        return modeName;
    }

    public void setModeName(String modeName) {
        this.modeName = modeName;
    }
}
