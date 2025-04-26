package com.hf.sfm.system.business;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.hibernate.Transaction;
import com.hf.sfm.system.pdo.Menu;
import com.hf.sfm.util.DaoFactory;

class MenuManage_saveOrUpdate_0_3_Test {

    private MenuManage menuManage;

    private Menu menu;

    @BeforeEach
    void setUp() {
        menuManage = mock(MenuManage.class);
        menu = new Menu();
    }

    @Test
    void testSaveOrUpdateNewMenu() {
        when(menuManage.saveOrUpdate(menu)).thenReturn("1");
        String result = menuManage.saveOrUpdate(menu);
        assertEquals("1", result);
        verify(menuManage).saveOrUpdate(menu);
    }

    @Test
    void testSaveOrUpdateExistingMenu() {
        menu.setIdno("123");
        when(menuManage.saveOrUpdate(menu)).thenReturn("1");
        String result = menuManage.saveOrUpdate(menu);
        assertEquals("1", result);
        verify(menuManage).update(menu);
    }

    @Test
    void testSaveOrUpdateException() {
        doThrow(new RuntimeException()).when(menuManage).saveOrUpdate(menu);
        String result = menuManage.saveOrUpdate(menu);
        assertEquals("0", result);
        verify(menuManage).saveOrUpdate(menu);
    }
}
