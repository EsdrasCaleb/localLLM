package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class ModeList_getMode_2_1_Test {

    @Test
    void testGetMode() {
        ModeList modeList = new ModeList();
        Mode mode1 = modeList.getMode("Mode1");
        Mode mode2 = modeList.getMode("Mode2");
        Mode mode3 = modeList.getMode("Mode3");
        assertEquals(mode1, modeList.getMode("Mode1"));
        assertEquals(mode2, modeList.getMode("Mode2"));
        assertNull(mode3);
    }

    @Test
    void testGetModeWithoutMatchingName() {
        ModeList modeList = new ModeList();
        Mode mode4 = modeList.getMode("Mode4");
        Mode mode5 = modeList.getMode("Mode5");
        assertEquals(mode4, modeList.getMode("Mode4"));
        assertEquals(mode5, modeList.getMode("Mode5"));
    }

    @Test
    void testGetModeWithNullList() {
        ModeList modeList = new ModeList();
        Mode mode6 = modeList.getMode(null);
        assertNull(mode6);
    }
}
