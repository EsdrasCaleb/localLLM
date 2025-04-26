package com.hf.sfm.util;

import org.junit.runner.RunWith;
import org.junit.platform.runner.JUnitPlatform;
import org.junit.platform.suite.api.SelectClasses;

@RunWith(value = JUnitPlatform.class)
@SelectClasses(value = { DaoFactory_rollback_4_0_Test.class, DaoFactory_encrypt_5_3_Test.class, DaoFactory_decrypt_6_0_Test.class, DaoFactory_update_8_0_Test.class, DaoFactory_closeAll_9_0_Test.class })
public class DaoFactory_Suite {
}
